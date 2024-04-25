import { minioClient } from '$lib/server/minioClient.js';

export const GET = async ({ url }) => {
	const bucketName = url.searchParams.get('bucketName');
	const fileName = url.searchParams.get('fileName');
	if (!bucketName || !fileName) {
		return new Response('Missing bucketName or fileName', { status: 400 });
	}
	try {
		const stream = await minioClient.getObject(bucketName, fileName);
		return new Response(stream as unknown as ReadableStream, {
			status: 200,
			headers: {
				'Content-Type': 'application/octet-stream'
			}
		});
	} catch (error) {
		console.error(error);
		return new Response('Error getting object', { status: 500 });
	}
};
