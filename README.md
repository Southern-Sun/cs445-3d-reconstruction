# cs445-3d-reconstruction

## Docker
Build:
`docker login`
`docker build -t johnsmall407/vggt-api:latest`
Push:
`docker push johnsmall407/vggt-api:latest`
Run locally:
`docker run --rm -p 8000:8000 -e PORT=8000 johnsmall407/vggt-api:latest`

## API Endpoint
`GET /health`
Simple health check that reports if the model is loaded

`curl https://xdf4mlzo4af10s.api.runpod.ai/health`

`POST /predict`
Form data `files` -> File to process

`curl -X POST https://xdf4mlzo4af10s.api.runpod.ai/predict -F "files=@C:/path/to/file"`

## API Testing
To run the api locally (no docker)
`uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
