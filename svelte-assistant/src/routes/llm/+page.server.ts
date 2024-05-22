import { LLM_BASE_URL } from '$env/static/private';

export const load = async () => {
	try {
		console.log("models url", `${LLM_BASE_URL}/api/tags`)
		const response = await fetch(`${LLM_BASE_URL}/api/tags`);
		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`);
		}
		const availableModels = await response.json();
		return {
			availableModels: availableModels.models
		};
	} catch (error) {
		console.error('Error fetching data:', error);
		return { availableModels: {} };
	}
};

