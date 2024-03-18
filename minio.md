# QuA: minio S3 server configuration

In this updated version, ingested documents are stored in a [minio](https://min.io/) instance.

## Requirements

It is assumed that `docker` is installed (tested on version 24.0.2).

## Start-up

To start ``minio``:

    docker run --name minio -d -p 9000:9000 -p 9001:9001 -v {local_storage_path}:/data quay.io/minio/minio server /data --console-address ":9001"

The minio server will be available on: http://localhost:9000

The minio server console will be available on: http://localhost:9001

The default username and password are:

    minioadmin
    minioadmin

Once this image is running, it is best to simply use

    docker stop minio
    docker start minio

to stop and start the minio service.

## Configuration

Log in to the minio console.

Under ``Settings: Configuration`` create a Server Location called ``ait``

Under ``Access Keys`` click the ``Create Access Key +`` button and confirm on the next page with the ``Create`` button.
Copy the resulting access key and secret key to the local ``config.ini`` file.