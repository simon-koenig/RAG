<script lang="ts">
	import MarqoResultCard from '$lib/components/MarqoResultCard.svelte';
	const { data } = $props();

	let defaultModel =
		data.availableModels.find((model: { model: string }) => model.model === 'llama3:latest') ||
		data.availableModels.find((model: { model: string }) => model.model === 'mixtral:latest') ||
		data.availableModels[0];
	let modelChoice = $state(defaultModel.model);
	let query = $state('');
	let background = $state('');
	let answer = $state('');
	let isFetching = $state(false);
	let sources = $state();





	const fetchCompletion = async () => {
		isFetching = true;
		answer = '';
		background ="default background";
		console.log("background: ", background);
		modelChoice = "mixtral";

		//
		// Marqo Lookup
		//
		try {
			const searchData = {
			q: query,
			searchableAttributes: ["Text"],
			searchMethod: 'TENSOR'
		};

		const response = await fetch("api/marqo", {
		  method: 'POST',
		  headers: {
			'Content-Type': 'application/json'
		  },
		  body: JSON.stringify({searchData, index: "qm-full"})
		});
  
		if (!response.ok) {
		  throw new Error('Network response was not ok');
		}
  
		const data = await response.json();
		console.log(data); // Handle the response data here
		handleSearchResults(data);
		sources = data; 
		} catch (error) {
			console.error('Error fetching data:', error);
		}
		//
		// LLM Response
		//
		try {
			const response = await fetch('/api/llm', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ query: query, background: background, model: modelChoice})
			});
			if (!response.body) {
				console.log('Failed to get readable stream');
				return;
			}
			const reader = response.body.getReader();
			const decoder = new TextDecoder();

			while (true) {
				const { done, value } = await reader.read();
				if (done) break;
				const chunk = decoder.decode(value, { stream: true });
				answer += chunk;
			}
		} catch (error) {
			console.error('Failed to fetch stream data:', error);
		} finally {
			isFetching = false;
		}
	};

	function handleSearchResults(results: { hits: string | any[]; }) {
		background = "";
		let numhits = results.hits.length;
		let durldict = {};
		let numtokens = query.length; // TODO: Add chatbot answer length to this
		const MAXTOKENS = 1000;
		const num_ref = 3;
		const query_threshold = 0.5;
		let temp_sources = [];
		if (numhits > 0) {
		let num_sources = 0;
		for (let i = 0; i < numhits && i < num_ref; i++) {
			let score = parseFloat(results.hits[i]._score);
			if (score >= query_threshold) {
			let fuid = results.hits[i].fuid;
			
			numtokens += results.hits[i].tokens;
			// let sourcetext = escapeMarkdown(results.hits[i].Text);
			let sourcetext = results.hits[i].Text;

			if (numtokens < MAXTOKENS) {
				let scorestring = score.toFixed(2);
				let refstring = `[${results.hits[i].Title}, page ${results.hits[i].Page}, paragraph ${results.hits[i].Paragraph}]\n`;
				background += sourcetext + " ";
				num_sources++;
			} else {
				console.log("Model token limit exceeded, sources reduced to " + i);
				break;
			}
			}
		}
		}
	}



	





</script>

<div class="flex gap-3 w-full justify-center items-end">
	<!-- <label class="text-md flex flex-col gap-1">
		Available Models:
		<select
			bind:value={modelChoice}
			class="border bg-rose-900/80 text-rose-100 px-2 py-1 rounded-lg"
		>
			{#each data.availableModels as model}
				<option value={model.model}>{model.name}</option>
			{/each}
		</select>
	</label> -->
	<label for="prompt" class="text-md flex flex-col gap-1"
		>Ask a Question/ Frag mich etwas!
		<input
			onkeypress={(e) => e.key === 'Enter' && fetchCompletion()}
			class="w-80 border px-2 py-1 rounded-lg"
			bind:value={query}
			spellcheck="false"
			name="prompt"
			type="text"
			autocomplete="off"
		/></label
	>

    
	<button
		onclick={fetchCompletion}
		disabled={isFetching || !query}
		class="disabled:opacity-100 h-fit w-[13ch] bg-rose-900 text-rose-100 rounded-xl py-1 px-5"
		>{isFetching ? 'Generating' : 'Submit'}</button
	>
	<!-- Abort button, not implemented yet -->
	<!--
    {#if isFetching}
		<button
			onclick={() => {
				prompt = '';
				answer = '';
			}}
			disabled={!isFetching}
			class="disabled:opacity-100 h-fit w-[13ch] bg-red-800 text-white rounded-xl py-1 px-5"
			>Cancel</button
		>
	{/if} -->
</div>

<div class="flex gap-4">
	<div class="w-1/2">
		<div class="mt-12 border rounded-lg p-5 mx-auto max-w-[800px]">
			<h2 class="border-b pb-1 text-rose-900">Sources</h2>
			<br />
			{#if sources}
			<div class="flex flex-wrap gap-4 mx-auto">
				{#each sources.hits as result (result._id)}
					<MarqoResultCard bucketName={"qm-full"} {result} />
				{/each}
			</div>
		{/if}

		</div>
	</div>
	<div class="w-1/2">
		<div class="mt-12 border rounded-lg p-5 mx-auto max-w-[800px]">
			<h2 class="border-b pb-1 text-rose-900">Response</h2>
			<br />
			{answer}
		</div>
	</div>
</div>
