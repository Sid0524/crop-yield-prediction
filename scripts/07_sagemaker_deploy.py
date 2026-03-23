"""
Phase 7 — AWS SageMaker Deployment
=====================================
Packages the trained XGBoost model and deploys it to a SageMaker
real-time inference endpoint.

Prerequisites:
  - AWS credentials configured (aws configure OR env vars)
  - IAM role with AmazonSageMakerFullAccess
  - Set SAGEMAKER_ROLE_ARN environment variable
  - S3 bucket writable by the role

If no AWS credentials are detected, the script runs in MOCK MODE:
it packages the model artifact and prints what the deployment would
do, but makes no AWS API calls. Safe for portfolio review.

Usage:
  export SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/SageMakerRole
  python scripts/07_sagemaker_deploy.py

Outputs:
  sagemaker/model.tar.gz
  sagemaker/deploy_config.json
"""

import os
import sys
import json
import tarfile
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

MODELS_DIR  = ROOT / "models"
SM_DIR      = ROOT / "sagemaker"
SM_DIR.mkdir(parents=True, exist_ok=True)

ENDPOINT_NAME   = "crop-yield-predictor-v1"
INSTANCE_TYPE   = "ml.m5.xlarge"     # 4 vCPU, 16 GB — cost ~$0.23/hr
FRAMEWORK_VER   = "2.0-1"            # XGBoost 2.x SageMaker container
BUCKET_PREFIX   = "crop-yield-models"


# ---------------------------------------------------------------------------
# Step 1: Package model artifact
# ---------------------------------------------------------------------------

def package_artifact() -> pathlib.Path:
    """
    Bundle xgboost_model.json + inference.py into model.tar.gz.
    SageMaker expects this layout inside the tarball:
      xgboost_model.json   — model weights
      inference.py         — entry point script
    """
    tar_path = SM_DIR / "model.tar.gz"
    model_json = MODELS_DIR / "xgboost_model.json"
    infer_py   = SM_DIR / "inference.py"

    if not model_json.exists():
        raise FileNotFoundError(
            f"{model_json} not found. Run 03_train_models.py first."
        )
    if not infer_py.exists():
        raise FileNotFoundError(
            f"{infer_py} not found. Expected in sagemaker/inference.py."
        )

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(str(model_json), arcname="xgboost_model.json")
        tar.add(str(infer_py),   arcname="inference.py")

    size_mb = tar_path.stat().st_size / 1e6
    print(f"[OK] Packaged model artifact -> sagemaker/model.tar.gz ({size_mb:.1f} MB)")
    return tar_path


# ---------------------------------------------------------------------------
# Step 2: Check AWS credentials
# ---------------------------------------------------------------------------

def get_aws_session():
    """Return a boto3 session if credentials exist, else None."""
    try:
        import boto3
        session = boto3.Session()
        creds   = session.get_credentials()
        if creds is None:
            return None
        creds.get_frozen_credentials()  # forces resolution (catches expired keys)
        return session
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Step 3: Deploy
# ---------------------------------------------------------------------------

def deploy_to_sagemaker(session, tar_path: pathlib.Path, role_arn: str):
    """Upload artifact and create/update SageMaker endpoint."""
    import boto3
    import sagemaker
    from sagemaker.xgboost import XGBoostModel

    sm_session = sagemaker.Session(boto_session=session)
    account_id = session.client("sts").get_caller_identity()["Account"]
    bucket     = f"{BUCKET_PREFIX}-{account_id}"

    # Ensure bucket exists
    s3 = session.client("s3")
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        region = session.region_name or "us-east-1"
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        print(f"  Created S3 bucket: s3://{bucket}")

    # Upload model artifact
    s3_key        = "models/xgboost_crop_yield/model.tar.gz"
    model_data_s3 = f"s3://{bucket}/{s3_key}"
    print(f"  Uploading to {model_data_s3} ...")
    s3.upload_file(str(tar_path), bucket, s3_key)
    print(f"  Upload complete.")

    # Create SageMaker model object
    xgb_sm_model = XGBoostModel(
        model_data=model_data_s3,
        role=role_arn,
        entry_point="inference.py",
        framework_version=FRAMEWORK_VER,
        py_version="py3",
        sagemaker_session=sm_session,
    )

    # Deploy real-time endpoint
    print(f"  Deploying endpoint '{ENDPOINT_NAME}' on {INSTANCE_TYPE} ...")
    print(f"  (This typically takes 5-8 minutes)")
    predictor = xgb_sm_model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
        serializer=sagemaker.serializers.JSONSerializer(),
        deserializer=sagemaker.deserializers.JSONDeserializer(),
    )
    print(f"  [OK] Endpoint live: {ENDPOINT_NAME}")

    # Smoke test
    import json as _json
    # [avg_temp_c, total_rain_mm, gdd, soil_quality, irrigation_frac,
    #  fertilizer_kg_ha, year]  — crop dummies and country_enc = 0 for test
    test_payload = [[24.0, 1083.0, 1820.0, 70.0, 0.40, 130.0, 2021,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5]]
    response = predictor.predict(test_payload)
    pred_yield = response["predictions"][0]
    print(f"  Smoke test passed — predicted yield: {pred_yield:.0f} kg/ha")

    return model_data_s3, predictor


# ---------------------------------------------------------------------------
# Mock mode output
# ---------------------------------------------------------------------------

def mock_deploy(tar_path: pathlib.Path):
    print("\n" + "=" * 60)
    print("  MOCK MODE — No AWS credentials detected")
    print("=" * 60)
    print("  Model artifact packaged successfully.")
    print(f"  Artifact: {tar_path}")
    print()
    print("  To deploy for real, run:")
    print("    export SAGEMAKER_ROLE_ARN=arn:aws:iam::<account>:role/<role>")
    print("    python scripts/07_sagemaker_deploy.py")
    print()
    print("  Deployment config that WOULD be used:")
    config = {
        "endpoint_name":   ENDPOINT_NAME,
        "instance_type":   INSTANCE_TYPE,
        "framework":       f"xgboost-{FRAMEWORK_VER}",
        "estimated_cost":  "~$0.23/hr (ml.m5.xlarge)",
        "delete_endpoint": "predictor.delete_endpoint()  # ALWAYS run after testing",
    }
    for k, v in config.items():
        print(f"    {k}: {v}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== SageMaker Deployment Script ===\n")

    # Package artifact (always runs — no AWS needed)
    tar_path = package_artifact()

    session  = get_aws_session()
    role_arn = os.environ.get("SAGEMAKER_ROLE_ARN", "")

    if session is None or not role_arn:
        mock_deploy(tar_path)
        # Save a placeholder deploy config
        config = {
            "mode":          "mock",
            "endpoint_name": ENDPOINT_NAME,
            "instance_type": INSTANCE_TYPE,
            "framework":     FRAMEWORK_VER,
            "artifact":      str(tar_path),
        }
    else:
        model_s3, predictor = deploy_to_sagemaker(session, tar_path, role_arn)

        with open(MODELS_DIR / "feature_names.json") as f:
            feature_names = json.load(f)

        config = {
            "mode":            "deployed",
            "endpoint_name":   ENDPOINT_NAME,
            "instance_type":   INSTANCE_TYPE,
            "framework":       FRAMEWORK_VER,
            "model_s3_uri":    model_s3,
            "feature_order":   feature_names,
            "cost_reminder":   "Delete endpoint when done: predictor.delete_endpoint()",
        }

        print("\n[IMPORTANT] Delete the endpoint when done to avoid charges:")
        print("  predictor.delete_endpoint()")

    with open(SM_DIR / "deploy_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nDeploy config saved -> sagemaker/deploy_config.json")


if __name__ == "__main__":
    main()
