<script lang="ts">
	import MarqoResultCard from '$lib/components/MarqoResultCard.svelte';

	const { data } = $props();

	let marqoIndex = $state(data.indexes.results[0].index_name ?? '');
	let query = $state('');
	let isFetching = $state(false);
	let promise: Promise<any> | null = $state(null);

	const mqLookUp = async () => {
		isFetching = true;
		const searchData = {
			q: query,
			searchableAttributes: ['Text'],
			searchMethod: 'TENSOR'
		};

		try {
			const response = await fetch('api/marqo', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ searchData, index: marqoIndex })
			});
			if (!response.ok) throw new Error('Network response was not ok');

			const data = await response.json();
			return data;
		} catch (error) {
			console.error('Error fetching data:', error);
		} finally {
			isFetching = false;
		}
	};

	const handleSubmit = () => {
		promise = mqLookUp();
	};
</script>

<div class="flex gap-3 w-full justify-center items-end">
	<label class="text-md flex flex-col gap-1">
		Index:
		<select bind:value={marqoIndex} class="border bg-rose-900/80 text-rose-100 px-2 py-1 rounded-lg">
			{#each data.indexes.results as index}
				<option value={index.index_name}>{index.index_name}</option>
			{/each}
		</select>
	</label>
	<label for="prompt" class="text-md flex flex-col gap-1"
		>Retrieve from Vector DB:
		<input
			onkeypress={(e) => e.key === 'Enter' && handleSubmit()}
			class="w-80 border px-2 py-1 rounded-lg"
			bind:value={query}
			spellcheck="false"
			name="prompt"
			type="text"
			autocomplete="off"
		/></label
	>
	<button
		onclick={handleSubmit}
		disabled={isFetching || !query}
		class="disabled:opacity-50 h-fit w-[13ch] bg-rose-900 text-rose-100 rounded-xl py-1 px-5"
		>Search</button
	>
</div>

<div class="mt-5">
	{#await promise}
		<div class="h-full w-full flex justify-center items-center">
			<p>Searching...</p>
		</div>
	{:then data}
		{#if data}
			<div class="flex flex-wrap gap-4 mx-auto">
				{#each data.hits as result (result._id)}
					<MarqoResultCard bucketName={marqoIndex} {result} />
				{/each}
			</div>
		{/if}
	{:catch error}
		<p style="color: red">{error.message}</p>
	{/await}
</div>
