# Use an official Python runtime as a parent image
FROM nvidia/cuda:12.2.0-base-ubuntu20.04

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y python3-pip

# Install required packages
RUN pip3 install -f requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define an environment variable
# This variable will be used by Uvicorn as the binding address
ENV HOST 0.0.0.0

# Run the FastAPI application using Uvicorn when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
