# For development

## Environments

```bash
make env
# or
pip install -r requirements.txt -q
flake8 --install-hook git
git config --bool flake8.strict true
git config --bool flake8.lazy true
```

