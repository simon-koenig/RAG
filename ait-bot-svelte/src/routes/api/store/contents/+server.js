import * as Minio from 'minio';
import { json } from 'stream/consumers';


// [INDEX]
let INDEX = "animal-facts"
// [MINIO]
let MINIO_ENDPOINT = "10.103.251.100" ;
let MINIO_ACCESS_KEY = "fZj3gEuMh9qdZ0Nz9UtQ" ;
let MINIO_SECRET_KEY = "un9y7C2KnlGmZ4tHgWS0fQBllqpcUnGCHoseD6cg" ;

// Initialize MinIO client
const minioClient = new Minio.Client({
    endPoint: MINIO_ENDPOINT,
    port: 9000,
    accessKey:  MINIO_ACCESS_KEY,
    secretKey: MINIO_SECRET_KEY,
    useSSL: false,
    region: "ait",
});
 

export async function POST(event){
    console.log("Entered Post to list all objects in a bucket.")
     // List all buckets
     console.log(event.request)
     try {
        let INDEX = await event.request.json();
        console.log(INDEX);
        const objects = minioClient.listObjects(INDEX,'');
        console.log(objects);
        return objects
     }
     catch(error)
     {
        console.error('Error accessing minio:', error);
     }
}