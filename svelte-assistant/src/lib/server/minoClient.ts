import * as Minio from 'minio';
import { MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_URL } from '$env/static/private';

export const minioClient = new Minio.Client({
	endPoint: MINIO_URL,
	port: 9000,
	useSSL: false,
	accessKey: MINIO_ACCESS_KEY,
	secretKey: MINIO_SECRET_KEY
});
