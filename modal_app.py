import modal

app = modal.App("damagevision-api")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "python-multipart==0.0.6",
        "tifffile==2023.9.26",
        "numpy==1.24.3",
        "opencv-python-headless==4.8.1.78",
        "Pillow==10.1.0",
        "ultralytics==8.0.200",
        "torch",
        "torchvision",
    )
)

@app.function(image=image, gpu="T4", timeout=600)
@modal.asgi_app()
def fastapi_app():
    from app import app as fastapi
    return fastapi
