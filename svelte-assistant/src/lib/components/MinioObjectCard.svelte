<script lang="ts">
	import type { BucketItem } from 'minio';

	const {
		object,
		bucketName
	}: {
		object: BucketItem;
		bucketName: string;
	} = $props();
</script>

<div
	class="max-w-sm rounded-lg overflow-hidden shadow-lg w-[40ch] flex-col justify-between p-4 border-[1px] boder-rose-200"
>
	<div class="flex flex-col gap-2">
		<div class="font-bold opacity-80 text-xl mb-2">{object.name}</div>
		{#if object.prefix}
			<p class="text-gray-700 text-base">
				Prefix: {object.prefix.toLocaleString()}
			</p>
		{/if}
		<p class="text-gray-700 text-base">
			Last Modified: {object.lastModified?.toLocaleString()}
		</p>
		<!-- 		<p class="text-gray-700 text-base">
			ETag: <br />
			{object.etag}
		</p> -->
		<p class="text-gray-700 text-base">
			Size: {object.size >= 1024 * 1024
				? `${(object.size / 1024 / 1024).toFixed(2)} MB`
				: `${Math.round(object.size / 1024).toLocaleString()} kB`}
		</p>
	</div>

	<div class="flex justify-end">
		<a
			class="bg-rose-700/80 px-2 py-1 rounded-lg text-rose-50 hover:bg-rose-900 hover:shadow-md"
			href={`${bucketName}/${object.name}`}
		>
			Preview
		</a>
	</div>
</div>
