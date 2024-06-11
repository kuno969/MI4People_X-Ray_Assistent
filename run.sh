#!/bin/sh
export PYTHONPATH=$PYTHONPATH:/mnt/code
. ./.env
# python test_api.py
streamlit run app.py
