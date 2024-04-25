<script lang="ts">
	const { children, data } = $props();
</script>

<div class="w-full flex justify-between">
	<div class="w-1/4">
		<div class="flex flex-col w-full gap-2 border-r">
			<h3 class="text-lg mb-3">Buckets</h3>
			{#each data.buckets as bucket}
				<a href={`/minio/${bucket.name}`} class="text-teal-700 hover:underline">{bucket.name}</a>
			{/each}
		</div>
	</div>
	<div class="w-full mx-12">
		{@render children()}
	</div>
</div>
