<script lang="ts">
	import '../app.pcss';
	import Header from '$lib/components/Header.svelte';

	let { children } = $props();
</script>

<svelte:head>
	<title>AIT Assistant Prototype</title>
</svelte:head>

<div class="min-h-dvh w-full px-20">
	<Header />
	<div class="mx-auto max-w-[1440px] mt-20">
		{@render children()}
	</div>
</div>
