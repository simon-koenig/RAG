<script lang="ts">
	import { page } from '$app/stores';
	import aitLogo from '$lib/images/ait-logo.png';

	const navlinks = [
		{ name: 'Assistant', href: '/llm' },
		{ name: 'Index Search', href: '/marqo' },
		{ name: 'Database', href: '/minio' }
	];
</script>

<div class="w-full h-20 border-b">
	<a href="https://www.ait.ac.at/en/" target="_blank">
		<img src={aitLogo} alt="AIT Logo" class="absolute top-0 left-0 h-10 w-25 m-4" />
	</a>
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

