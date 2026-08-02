# Deployment Info — Incident Remediation AI on AWS EKS

This document records the complete process used to take the Incident Remediation AI
service from local/offline development to a live deployment on Amazon EKS, backed by
real AWS Bedrock (LLM + embeddings) and a Qdrant Cloud vector store, with a GitHub
Actions CI/CD pipeline deploying to it automatically.

Steps are listed in the order they were actually performed. Each step states its
**purpose**, the **resources created or modified**, and the **exact commands** used.

---

## Environment at a glance

| Item               | Value                                                                    |
| ------------------ | ------------------------------------------------------------------------ |
| AWS Account        | `646966486357`                                                           |
| AWS Region         | `us-east-2`                                                              |
| EKS cluster name   | `incident-ai-demo`                                                       |
| Node instance type | `m7i-flex.large` (2 vCPU / 8 GiB, free-tier eligible)                    |
| Node count         | 2                                                                        |
| ECR repository     | `incident-remediation-ai`                                                |
| Vector store       | Qdrant Cloud (`incident_knowledge` collection)                           |
| LLM                | Bedrock inference profile `us.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| Embedding model    | `amazon.titan-embed-text-v2:0`                                           |
| GitHub repo        | `srividyap98/incident-remediation-ai` (private)                          |

---

## Phase 1 — Verify local AWS credentials and Bedrock access

**Purpose:** Confirm the machine actually had working AWS credentials before building
anything on top of them, and find out which Bedrock model IDs were genuinely
invokable in this account/region (the model ID configured in `.env.example` turned
out not to work as-is — see below).

**Resources touched:** none created; read-only checks against the existing IAM user.

```bash
aws sts get-caller-identity
aws configure get region
aws bedrock list-foundation-models --region us-east-2 \
  --query "modelSummaries[?contains(modelId,'claude') || contains(modelId,'titan-embed')].modelId"
```

**Finding:** Titan embeddings (`amazon.titan-embed-text-v2:0`) invoke directly, but
Claude Sonnet 4.5 does not — newer Claude models on Bedrock require an **inference
profile** rather than the bare model ID:

```bash
aws bedrock list-inference-profiles --region us-east-2 \
  --query "inferenceProfileSummaries[?contains(inferenceProfileId,'sonnet-4-5')]"
```

This returned `us.anthropic.claude-sonnet-4-5-20250929-v1:0`, which was confirmed
working with a direct `converse` call via `boto3`.

---

## Phase 2 — Switch the app from offline mode to real backends

**Purpose:** Move off the local FAISS + mocked-Bedrock offline dev setup and onto the
real Qdrant Cloud cluster and real Bedrock, so the deployed service would do genuine
retrieval and generation instead of canned responses.

**Resources modified:** local `.env` file only (not committed — contains secrets).

Key `.env` changes:

```dotenv
AWS_REGION=us-east-2
BEDROCK_LLM_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
VECTOR_STORE_BACKEND=qdrant
QDRANT_URL=https://<cluster-id>.us-west-1-0.aws.cloud.qdrant.io
QDRANT_API_KEY=<qdrant api key>
QDRANT_COLLECTION=incident_knowledge
OFFLINE_MODE=false
```

**Code fix — `rag/vector_store/factory.py`:** `langchain_community.vectorstores.Qdrant`
no longer exists in the installed LangChain version; the real integration lives in the
standalone `langchain-qdrant` package. Installed it and rewrote `_build_qdrant_store()`
to use `QdrantVectorStore`, auto-creating the collection (with the correct vector
dimension, probed at runtime from the embedding model) if it doesn't already exist:

```bash
pip install langchain-qdrant
```

Also added `langchain-qdrant>=0.2.0` to `pyproject.toml` dependencies so it isn't just
an ad-hoc venv install.

**Re-ran ingestion against the real backends:**

```bash
python -m rag.ingestion.pipeline --source ./data/sample_runbooks/
```

This embedded all 37 runbook chunks via real Bedrock Titan and upserted them into the
Qdrant Cloud collection (verified via `qdrant_client.get_collection()` → 37 points,
1024-dim vectors).

---

## Phase 3 — Initialize git and create the GitHub repository

**Purpose:** The project had no git history at all. A real git repo hosted on GitHub
is a prerequisite for GitHub Actions CI/CD.

**Resources created:** local `.git` repo; GitHub repo `srividyap98/incident-remediation-ai` (private).

```bash
git init
# .gitignore added first (.env, .venv/, node_modules/, data/, caches, etc.)
git add -A
git commit -m "Initial commit: multi-agent incident remediation AI"

# Authenticate the GitHub CLI (interactive device-code flow)
gh auth login --hostname github.com --git-protocol https --web

# Create the repo from the current directory and push
gh repo create incident-remediation-ai --private --source=. --remote=origin --push
```

---

## Phase 4 — Create the ECR repository

**Purpose:** A container registry to hold the built application image, for both the
manual first deploy and every subsequent CD-pipeline deploy.

**Resources created:** ECR repository `incident-remediation-ai` in `us-east-2`.

```bash
aws ecr create-repository \
  --repository-name incident-remediation-ai \
  --region us-east-2 \
  --image-scanning-configuration scanOnPush=true
```

Result URI: `646966486357.dkr.ecr.us-east-2.amazonaws.com/incident-remediation-ai`

---

## Phase 5 — Fix and build the Docker image

**Purpose:** Produce a working container image to run on EKS.

**Bug found — `Dockerfile` builder stage:** it ran `pip install --prefix=/install .`
right after copying only `pyproject.toml`, before any of the project's own source
directories existed in the build context. Since the project's own package (per
`[tool.setuptools.packages.find]`) has to be built from real source, this failed.
Fixed by copying `agents/`, `api/`, `rag/`, `config/`, `monitoring/` into the builder
stage _before_ the `pip install` step, and installing the `.[offline]` extra so the
image could also run in offline/local-embedding mode if needed.

**Bug found — too many uvicorn workers:** the Dockerfile's `CMD` ran
`uvicorn ... --workers 4`. Each worker loads the full dependency stack
(langchain, boto3, numpy, pandas, mlflow, opentelemetry, …); four of them
comfortably exceeded any reasonable container memory limit and caused persistent
OOM kills once deployed (see Phase 8). Changed to `--workers 1` — horizontal scaling
is handled by Kubernetes pod replicas instead, which is the idiomatic way to add
capacity in this setup.

**Build and push:**

```bash
aws ecr get-login-password --region us-east-2 | \
  docker login --username AWS --password-stdin 646966486357.dkr.ecr.us-east-2.amazonaws.com

docker build --platform linux/amd64 \
  -t 646966486357.dkr.ecr.us-east-2.amazonaws.com/incident-remediation-ai:latest .

docker push 646966486357.dkr.ecr.us-east-2.amazonaws.com/incident-remediation-ai:latest
```

---

## Phase 6 — Provision the EKS cluster

**Purpose:** Stand up a real Kubernetes cluster to run the service.

**Resources created:** `infra/eks/cluster.yaml` (eksctl cluster config, committed to
the repo); EKS cluster `incident-ai-demo` with OIDC enabled (`iam.withOIDC: true`,
required for IRSA in Phase 7); a managed node group.

Initial config used `t3.medium` nodes — **this failed**:

```
InvalidParameterCombination - The specified instance type is not eligible for Free Tier.
```

This AWS account is restricted to free-tier-eligible instance types only. Checked
which types actually qualify:

```bash
aws ec2 describe-instance-types --region us-east-2 \
  --filters "Name=free-tier-eligible,Values=true" \
  --query "InstanceTypes[].{Type:InstanceType,vCPU:VCpuInfo.DefaultVCpus,MemMiB:MemoryInfo.SizeInMiB}"
```

Retried with `t3.small` (2 GiB RAM) — the node group came up, but application pods
were repeatedly `OOMKilled` (exit code 137) even after tuning memory limits up to the
practical ceiling of that instance size (~1.4 GiB allocatable per node). Deleted that
node group and re-created it with `m7i-flex.large` (2 vCPU / 8 GiB, also flagged
free-tier-eligible by AWS's own API) — this resolved the memory pressure entirely.

```bash
eksctl create cluster -f infra/eks/cluster.yaml   # first attempt, t3.medium — failed
# edited cluster.yaml -> instanceType: t3.small, retried, nodegroup created but pods OOM'd
eksctl delete nodegroup --cluster incident-ai-demo --name ng-default --region us-east-2 --drain=false
# edited cluster.yaml -> instanceType: m7i-flex.large
eksctl create nodegroup -f infra/eks/cluster.yaml   # succeeded

aws eks update-kubeconfig --name incident-ai-demo --region us-east-2
kubectl get nodes -o wide
```

Final `infra/eks/cluster.yaml`:

```yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: incident-ai-demo
  region: us-east-2
  version: "1.31"
managedNodeGroups:
  - name: ng-default
    instanceType: m7i-flex.large
    desiredCapacity: 2
    minSize: 2
    maxSize: 2
    volumeSize: 20
iam:
  withOIDC: true
```

---

## Phase 7 — IAM roles for pod-level Bedrock access (IRSA)

**Purpose:** Let the application pods call Bedrock using a scoped IAM role assumed
via their Kubernetes service account — no AWS access keys baked into the container.

**Resources created:** IAM role `incident-ai-bedrock-role`, trusted by the cluster's
own OIDC provider, scoped to the `incident-ai-sa` service account in the `incident-ai`
namespace.

```bash
# Get the cluster's OIDC issuer
aws eks describe-cluster --name incident-ai-demo --region us-east-2 \
  --query "cluster.identity.oidc.issuer"

# Trust policy scoped to system:serviceaccount:incident-ai:incident-ai-sa
aws iam create-role \
  --role-name incident-ai-bedrock-role \
  --assume-role-policy-document file://irsa-trust-policy.json

aws iam put-role-policy \
  --role-name incident-ai-bedrock-role \
  --policy-name bedrock-invoke \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      "Resource": "*"
    }]
  }'
```

This role ARN is referenced by the `ServiceAccount` manifest in
`infra/eks/deployment.yaml` (see Phase 8).

---

## Phase 8 — Deploy the application to Kubernetes

**Purpose:** Actually run the service on the cluster.

**Resources created:** `incident-ai` namespace; `incident-ai-config` ConfigMap;
`incident-ai-secrets` Secret (Qdrant API key); `incident-ai-api` Deployment, Service,
and HorizontalPodAutoscaler; `incident-ai-sa` ServiceAccount (IRSA-annotated).

```bash
kubectl create namespace incident-ai

kubectl create secret generic incident-ai-secrets \
  --namespace incident-ai \
  --from-literal=QDRANT_API_KEY='<qdrant api key>'

kubectl apply -f infra/eks/deployment.yaml
```

`infra/eks/deployment.yaml` placeholders (`<AWS_ACCOUNT_ID>`, `<REGION>`) were filled
in with real values (account `646966486357`, region `us-east-2`, the ECR image URI,
and the IRSA role ARN from Phase 7). The ConfigMap was also extended with
`BEDROCK_LLM_MODEL_ID`, `BEDROCK_EMBEDDING_MODEL_ID`, `QDRANT_URL`, and
`QDRANT_COLLECTION`, none of which were present in the original template.

**Debugging pod instability (three separate causes found and fixed in sequence):**

1. **OOMKilled (exit 137) on `t3.small`.** Root cause: 4 uvicorn workers per pod
   plus a tight 900Mi memory limit. Fixed the Dockerfile (Phase 5) and raised the
   limit; still OOM'd because the _node_ itself only had ~1.4 GiB allocatable —
   ultimately fixed by moving to `m7i-flex.large` nodes (Phase 6).
2. **Liveness probe timeouts** (`context deadline exceeded`) causing repeated
   kill-and-restart cycles. The default 1-second probe timeout was too aggressive for
   a CPU-throttled burstable instance under load. Fixed by adding `timeoutSeconds: 8`
   and raising `failureThreshold` to 5 on both probes.
3. **Stale replicas holding node capacity** during rollouts — force-deleted old
   `ContainerStatusUnknown`/`Error` pods left over from earlier failed rollouts so the
   scheduler could place fresh pods.

Final resource block used in `infra/eks/deployment.yaml`:

```yaml
resources:
  requests:
    cpu: "250m"
    memory: "512Mi"
  limits:
    cpu: "1"
    memory: "2Gi"
livenessProbe:
  httpGet: { path: /health, port: 8000 }
  initialDelaySeconds: 30
  periodSeconds: 15
  timeoutSeconds: 8
  failureThreshold: 5
readinessProbe:
  httpGet: { path: /health, port: 8000 }
  initialDelaySeconds: 10
  periodSeconds: 10
  timeoutSeconds: 8
  failureThreshold: 5
```

**Verification once stable:**

```bash
kubectl get pods -n incident-ai -o wide
kubectl rollout status deployment/incident-ai-api -n incident-ai
# READY 2/2, 0 restarts
```

---

## Phase 9 — CI/CD pipeline (GitHub Actions)

**Purpose:** Automate lint/test/build on every push, and automate build → push → deploy
on every push to `main`, using short-lived OIDC credentials instead of stored AWS keys.

**Resources created:**

- `.github/workflows/ci.yml` — ruff, mypy (non-blocking), pytest (unit + integration,
  `OFFLINE_MODE=true` so no AWS/Qdrant secrets needed for CI), a Docker build-check job,
  and a TypeScript client type-check job.
- `.github/workflows/cd.yml` — on push to `main`: assumes an AWS IAM role via OIDC,
  logs into ECR, builds and pushes the image (tagged with the git SHA and `latest`),
  updates the deployment's image, and waits for rollout.
- IAM OIDC provider for `token.actions.githubusercontent.com`.
- IAM role `incident-ai-github-actions`, trusted only for this specific repo, with
  permissions scoped to `ecr:GetAuthorizationToken` / push actions on this one ECR
  repo and `eks:DescribeCluster` on this one cluster.
- An EKS **access entry** granting that role's principal actual Kubernetes RBAC
  (`AmazonEKSEditPolicy`, scoped to the `incident-ai` namespace) — IAM permissions
  alone do not grant `kubectl` access; that is a separate authorization layer.

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 1c58a3a8518e8759bf075b76b750d4f2df264fcd

aws iam create-role \
  --role-name incident-ai-github-actions \
  --assume-role-policy-document file://gha-trust-policy.json

aws iam put-role-policy \
  --role-name incident-ai-github-actions \
  --policy-name ecr-push-eks-describe \
  --policy-document file://gha-permissions-policy.json

aws eks create-access-entry \
  --cluster-name incident-ai-demo --region us-east-2 \
  --principal-arn arn:aws:iam::646966486357:role/incident-ai-github-actions \
  --type STANDARD

aws eks associate-access-policy \
  --cluster-name incident-ai-demo --region us-east-2 \
  --principal-arn arn:aws:iam::646966486357:role/incident-ai-github-actions \
  --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSEditPolicy \
  --access-scope type=namespace,namespaces=incident-ai
```

**Debugging the pipeline itself (three issues found via real pushed runs):**

1. **CI lint failure that didn't reproduce locally.** CI did a fresh
   `pip install -e ".[dev]"` and got `ruff 0.16.1`, a newer version than the locally
   tested `0.15.22` — the newer release had promoted ~50 additional rules into its
   default set. Fixed by pinning `ruff==0.15.22` in `pyproject.toml` so CI and local
   stay deterministic regardless of upstream releases.
2. **CD's OIDC role assumption failed** (`Not authorized to perform
sts:AssumeRoleWithWebIdentity`) even though the trust policy looked correct.
   Pulled the actual rejected request from CloudTrail and found GitHub's real OIDC
   subject claim was `repo:srividyap98@51413023/incident-remediation-ai@1318882670:ref:refs/heads/main`
   — it embeds numeric owner/repo IDs after `@`, not the classic documented
   `repo:OWNER/REPO:ref:...` format. Updated the trust policy's `sub` condition to
   `repo:srividyap98*/incident-remediation-ai*:*` to match.
3. **`kubectl rollout status` timing out** even though the deployment was actually
   healthy — `maxSurge: 1` forced strictly sequential pod replacement, and each new
   pod's ~3.2 GB image pull pushed total rollout time past first a 180s, then a 400s
   timeout. Raised `maxSurge` to `2` (real headroom exists on `m7i-flex.large` nodes)
   and the workflow timeout to `600s`.

```bash
git add -A && git commit -m "Add CI/CD pipeline: GitHub Actions -> ECR -> EKS"
git push origin main
gh run list --repo srividyap98/incident-remediation-ai
gh run view <run-id> --log-failed   # used repeatedly to debug the three issues above
```

---

## Phase 10 — Validate the deployed application

**Purpose:** Confirm the live cluster actually serves correct, real (non-mocked)
results before considering the deployment successful.

**Commands used:**

```bash
# Health check via port-forward
kubectl port-forward -n incident-ai svc/incident-ai-api 8080:80 &
curl http://localhost:8080/health
# {"status":"ok","vector_store_backend":"qdrant","bedrock_model":"us.anthropic.claude-sonnet-4-5-20250929-v1:0"}

# Real incident submission through the live pods
curl -X POST http://localhost:8080/api/v1/incidents/ \
  -H "Content-Type: application/json" \
  -d @sample_incident.json
```

**Result confirmed:**

- `agent_messages` showed real semantic retrieval from Qdrant (`"Found 5 relevant
documents (top score: 0.709)"`), not the offline mock's fixed responses.
- `root_cause_analysis` and `remediation_steps` were genuinely incident-specific
  (e.g., Elasticsearch `match_all` query and heap-usage details for a search-latency
  incident, PgBouncer/`pg_terminate_backend` detail for a connection-pool incident) —
  grounded in the actual retrieved runbook content via real Bedrock reasoning.
- A full end-to-end CD run (push → CI → build → push to ECR → deploy → rollout) was
  also verified successful, with the resulting pods passing the same health check.

This confirmed the deployment — infrastructure, application, and pipeline — was fully
functional before the cluster was torn down to stop further AWS charges.

---

Your turn — commit, branch, and open the PR
Since the goal is practice, do this part yourself rather than me pushing directly:

cd "/Users/srividyapanchagnula/Downloads/incident-remediation-ai 2"

# 1. Create a feature branch (never commit fixes straight to main)

git checkout -b fix/lambda-handler-types

# 2. Stage and commit just this file

git add infra/lambda/lambda_handler.py
git commit -m "Fix missing type annotations in lambda_handler.py"

# 3. Push the branch to GitHub

git push -u origin fix/lambda-handler-types

# 4. Open the PR (gh is already authenticated in this session)

gh pr create --title "Fix missing type annotations in lambda_handler.py" \
 --body "Adds proper generics (dict[str, Any]) and a TYPE_CHECKING-guarded IncidentState import to resolve mypy strict-mode errors in the Lambda handler." \
 --base main

# 5. (Optional) watch CI run against your PR

gh pr checks --watch
After that, gh pr view --web opens it in the browser so you can see the diff, the CI checks running, and merge it yourself once green.
