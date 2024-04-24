<script lang="ts">
	import { goto } from '$app/navigation';

	let {
		result,
		bucketName
	}: {
		result: {
			Title: string;
			Text: string;
			Page: number;
			Paragraph: number;
			_highlights: {
				Text: string;
			};
			_score: number;
			tokens: number;
			fuid: string;
		};
		bucketName: string;
	} = $props();
	const { Title, Text, Page, Paragraph, _highlights, _score, tokens } = result;

	let index = Text.indexOf(_highlights.Text);
	let beforeHighlight = Text.substring(0, index);
	let highlight = _highlights.Text;
	let afterHighlight = Text.substring(index + _highlights.Text.length);
</script>

<article
	class="border flex flex-col mx-auto justify-between rounded-lg p-4 shadow-lg max-w-md my-4"
>
	<div class="pt-4">
		<div class="flex justify-between items-end pr-2">
			<h2 class="text-lg font-bold">{Title}</h2>
			<button
				onclick={() => window.open('/minio/' + bucketName + '/' + result.fuid, '_blank')}
				class="opacity-70 border px-2 py-1 rounded-lg hover:bg-rose-100"
			>
				⇒ View Document
			</button>
		</div>
		<p class="text-sm mt-2 text-gray-500">Page: {Page}, Paragraph: {Paragraph}</p>
		<div class="mt-4">
			<p class="text-gray-700 whitespace-pre-line break-words">
				{beforeHighlight}
				<span class="bg-yellow-200">{highlight}</span>
				{afterHighlight}
			</p>
		</div>
	</div>
	<div class="flex justify-between items-center mt-4">
		<span class="text-gray-600 text-xs">Tokens: {tokens}</span>
		<span class="text-gray-600 text-xs">Score: {_score.toFixed(2)}</span>
	</div>
</article>
