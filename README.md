# cs445-3d-reconstruction

## Docker
Build:
`docker login`
`docker build -t johnsmall407/vggt-api:latest`
Push:
`docker push johnsmall407/vggt-api:latest`
Run locally:
`docker run --rm -p 8000:8000 -e PORT=8000 johnsmall407/vggt-api:latest`

Tag:
`docker tag johnsmall407/vggt-api:latest johnsmall407/vggt-api:v0.6`

## API Endpoint
`GET /ping`
Simple health check that reports if the container is healthy

`curl https://f11nyung51hib2.api.runpod.ai/ping`

`POST /reconstruct`
Form data `files` -> File to process

`curl -X POST "https://f11nyung51hib2.api.runpod.ai/reconstruct" -F "files=@C:/<your_file>" -H "Authorization: Bearer <token>" --output output.glb`

## API Testing
To run the api locally (no docker)
`uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`

## API Wrapper
Run the API wrapper script to interact with the containers easily from the command line:
`python app/wrapper/vggt-api.py`

By passing in different verbs, the CLI tool will perform different actions, including:
- `mask`: apply a mask to all images in a directory to exclude unwanted image data
- `convert`: convert a video file to a set of image keyframes
- `reconstruct`: call the API with a set of image files and a confidence value of your choice and save the returned GLB file.
