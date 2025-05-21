name: Docker Image CI for Distil-Whisper

on:
  push:
    branches: [ "main", "master" ]
  pull_request:
    branches: [ "main", "master" ]
  workflow_dispatch:

jobs:
  build-and-push:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: List files in workspace # <<< --- ADD THIS DEBUGGING STEP
        run: ls -la
        # This command will list all files and directories, including hidden ones,
        # in the root of your checked-out repository within the GitHub Actions runner.

      - name: Maximize build space
        uses: easimon/maximize-build-space@v10
        with:
          root-reserve-mb: 5120
          swap-size-mb: 1024
          remove-dotnet: 'true'
          remove-android: 'true'
          remove-haskell: 'true'
          remove-codeql: 'true'

      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Docker Hub
        if: github.event_name != 'pull_request' && (github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master')
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Docker meta
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ secrets.DOCKERHUB_USERNAME }}/distil-whisper-runpod
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix=sha-
            type=raw,value=latest,enable={{is_default_branch}}
          flavor: |
            latest=auto

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile # This path is relative to the context
          push: ${{ github.event_name != 'pull_request' && (github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master') }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
