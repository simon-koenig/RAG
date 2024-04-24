import { openAI, prompt } from '$lib/server/llmClient';

async function chatComplete({
	query,
	background,
	model,
	systemPrompt = prompt,
	temperature = 0.5
}: {
	query: string;
	background: string;
	model: string;
	systemPrompt?: string;
	temperature?: number;
}) {
	return openAI.chat.completions.create({
		model,
		messages: [
			{ role: 'user', content: systemPrompt },
			{ role: 'assistant', content: background },
			{ role: 'user', content: query }
		],
		temperature,
		stream: true
	});
}

export const POST = async ({ request }) => {
	try {
		const { query, background, model} = await request.json();
		console.log('query:', query);
		console.log('background:', background);
		console.log('model:', model);

		const dataStream = await chatComplete( {query, background, model});

		const webStream = new ReadableStream({
			async start(controller) {
				for await (const chunk of dataStream) {
					const content = chunk.choices[0]?.delta?.content || '';
					controller.enqueue(content);
				}
				controller.close();
			}
		});

		return new Response(webStream, {
			headers: { 'Content-Type': 'text/plain' }
		});
	} catch (error) {
		console.error('Error processing chat completion:', error);
		return new Response(JSON.stringify({ error: 'Failed to process chat completion' }), {
			status: 500,
			headers: { 'Content-Type': 'application/json' }
		});
	}
};

