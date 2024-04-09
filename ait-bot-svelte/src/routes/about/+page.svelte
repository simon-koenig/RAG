<script>
	export let pdfUrl = "/Bengal-tiger-facts.pdf";
</script>

<svelte:head>
	<title>About</title>
	<meta name="description" content="About this app" />
</svelte:head>

<div class="text-column">
	<h1>About this app</h1>

	<p>
		This is a question answer system based on an open Source LLM and a RAG pipeline using a vector storage to fetch related data.
	</p>


	<p>
		The <a href="/recommendations">User recommendations</a> page illustrates how to achieve optimal results. 
	</p>

	<div class="pdf-container">
		<object type="application/pdf" data={pdfUrl} width="100%" height="100%">
		  <p>Sorry, your browser doesn't support embedded PDFs.</p>
		</object>
	</div>
</div>

<style>
	.pdf-container {
	  width: 100%;
	  height: 100vh;
	}
  </style>