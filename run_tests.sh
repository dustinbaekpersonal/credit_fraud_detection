#!/bin/bash
.venv/bin/python3 -m pytest tests -v --cov=src --cov-report xml:coverage.xml
