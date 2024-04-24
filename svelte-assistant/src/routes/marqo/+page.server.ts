import { MARQO_URL } from '$env/static/private';

export const load = async () => {
	const indexEndpoint = `${MARQO_URL}/indexes`;

	try {
		const response = await fetch(indexEndpoint);
		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`);
		}
		const indexes = await response.json();
		return { indexes };
	} catch (error) {
		console.error('Error fetching data:', error);
		return { indexes: {} };
	}
};
