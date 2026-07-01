set -euo pipefail

# Must run from inside the project venv so mlflow and other deps are available.
# Run: source .venv/bin/activate && bash run-airflow-standalone.sh

export AIRFLOW_HOME=~/airflow
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=false

mkdir -p $AIRFLOW_HOME

echo '{"admin": "admin"}' > $AIRFLOW_HOME/simple_auth_manager_passwords.json.generated

# Load .env if present
if [ -f .env ]; then
  set -a && source .env && set +a
fi

airflow standalone
