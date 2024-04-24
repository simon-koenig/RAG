<script lang="ts">
	import { page } from '$app/stores';

	const navlinks = [
		{ name: 'Assistant', href: '/llm' },
		{ name: 'Index Search', href: '/marqo' },
		{ name: 'Database', href: '/minio' }
	];
</script>

<div class="w-full h-20 border-b">
	<div class="flex h-full items-center justify-between mx-auto max-w-[1440px]">
		<a href="/"><h3> AIT Assistant Prototype </h3></a>
		{@render navbar()}
	</div>
</div>

{#snippet navbar()}
	<div class="flex gap-3 justify-center mx-4">
		{#each navlinks as { name, href }}
			<a
				{href}
				class:underline={$page.url.pathname.startsWith(href)}
				class="text-rose-900 hover:underline"
			>
				{name}
			</a>
		{/each}
	</div>
{/snippet}

