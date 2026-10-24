.PHONY: compare api ui stress ingest ask batch agent eval test runs
export PYTHONPATH := src
ingest:   ; python -m capstone.cli ingest --reset
ask:      ; python -m capstone.cli ask "$(Q)"
batch:    ; python -m capstone.cli batch eval/questions.txt
agent:    ; python -m capstone.cli agent "$(Q)"
eval:     ; python -m capstone.cli eval
test:     ; BACKEND=fake python -m pytest -q
runs:     ; bash scripts/w1_runs.sh
api:     ; uvicorn capstone.api:app --port 8000
ui:      ; streamlit run ui/app.py
stress:  ; python scripts/stress_test.py
compare: ; python scripts/compare_models.py
