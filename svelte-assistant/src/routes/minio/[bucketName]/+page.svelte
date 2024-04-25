<script lang="ts">
	import MinioObjectCard from '$lib/components/MinioObjectCard.svelte';

	let { data } = $props();
</script>

<h1 class="text-xl border-b pb-3">List of Files in "{data.bucketName}":</h1>

<div class="flex flex-wrap gap-3 mt-6">
	{#await data.streamed.objects}
		Loading bucket information...
	{:then objects}
		{#each objects as object (object.etag)}
			<MinioObjectCard {object} bucketName={data.bucketName} />
		{/each}
	{/await}
</div>
