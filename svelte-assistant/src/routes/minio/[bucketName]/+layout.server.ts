import { minioClient } from '$lib/server/minioClient';
import type { BucketItem } from 'minio';

export const load = async ({ params }) => {
	const { bucketName } = params;
	const data: Array<BucketItem> = [];
	const objects: Promise<Array<BucketItem>> = new Promise((resolve, reject) => {
		try {
			const stream = minioClient.listObjectsV2(bucketName);
			stream.on('data', (obj) => {
				data.push(obj);
			});
			stream.on('end', () => {
				resolve(data);
			});
			stream.on('error', (error) => {
				console.error(error);
				reject(error);
			});
		} catch (error) {
			console.error(error);
			reject(error);
		}
	});
	return {
		bucketName,
		streamed: { objects }
	};
};
