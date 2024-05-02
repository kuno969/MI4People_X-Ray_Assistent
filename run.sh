#!/bin/sh
export PYTHONPATH=$PYTHONPATH:/mnt/code
. ./.env
streamlit run app.py
