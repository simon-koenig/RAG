import { minioClient } from '$lib/server/minioClient';

export const load = async () => {
	try {
		const buckets = await minioClient.listBuckets();
		return { buckets };
	} catch (error) {
		console.error('Error listing buckets:', error);
		return { buckets: [] };
	}
};
