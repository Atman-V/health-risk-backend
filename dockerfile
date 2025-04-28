# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Debugging step: List the contents of /app to ensure requirements.txt is there
RUN ls -l /app

# Install pip, upgrade it, and install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that FastAPI will run on
EXPOSE 80

# Run gunicorn with uvicorn workers to serve the FastAPI app
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:80"]
