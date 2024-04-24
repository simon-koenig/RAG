import { MARQO_URL } from '$env/static/private';

export async function POST({ request, fetch }) {
	const { searchData, index } = await request.json();

	const searchEndpoint = `${MARQO_URL}/indexes/${index}/search`;

	try {
		const response = await fetch(searchEndpoint, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify(searchData)
		});

		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`);
		}
		return response;
	} catch (error) {
		console.error(error);
		return new Response('Error getting object', { status: 500 });
	}
}
