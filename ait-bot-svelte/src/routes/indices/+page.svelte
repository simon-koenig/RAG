<script>
	import { json } from '@sveltejs/kit';
	import { onMount } from 'svelte';
	import * as Minio from 'minio'

	/** @type {import('./$types').PageData} */
	let background = "";
	let query = '';
	let OPENAIMODEL = "mixtral";
	let ENDPOINT = "http://10.103.251.104:8040/v1";
	let MARQO_ENDPOINT = "http://10.103.251.100:8882";
	let APIKEY = "N/A";
	let INDEX = "animal-facts"
	// Define user prompt 
	let prompt = "You are designed to be helpful while providing only factual information. If you are uncertain, state it and explain why. Give an answer based on information in the following paragraphs."
  

    // [MINIO]
	let MINIO_ENDPOINT = "10.103.251.100:9000" ;
	let MINIO_ACCESS_KEY = "fZj3gEuMh9qdZ0Nz9UtQ" ;
	let MINIO_SECRET_KEY = "un9y7C2KnlGmZ4tHgWS0fQBllqpcUnGCHoseD6cg" ;

	// Initialize MinIO client
	const minioClient = new Minio.Client({
		endPoint: MINIO_ENDPOINT,
		accessKey:  MINIO_ACCESS_KEY,
		secretKey: MINIO_SECRET_KEY,
		region: "ait",
	});

  let objects = []; // To store the list of objects

  // Function to list objects in a bucket
  async function listObjects() {
    try {
      const objectsList = await minioClient.listObjects(INDEX, ''); // Provide your bucket name
      objects = objectsList.map(obj => obj.name);
    } catch (error) {
      console.error('Error listing objects:', error);
    }
  }

  // Call listObjects function when component mounts
  onMount(listObjects);
	
  </script>
  
	<svelte:head>
		<title>Simons Chatbot</title>
	</svelte:head>
  
  <main>
	<h1> Indices </h1>
	<div class="container">
		<div class="box">
		Choose Index here. Current index = {INDEX}
		</div>

		<div class="box">
			{#each objects as object}
			<div><strong>Source:</strong> {object}</div>
			{/each}
		</div>
	</div>


  </main>
  
  
  
  
  
  <style>
	  main {
		  text-align: center;
		  padding: 1em;
		  max-width: 240px;
		  margin: 0 auto;
	  }
  
	textarea {
	  width: 400px;
	}
  
	  h1 {
		  color: #000;
		  text-transform: uppercase;
		  font-size: 4em;
		  font-weight: 100;
	  }
  
	  @media (min-width: 640px) {
		  main {
			  max-width: none;
		  }
	  }

	.container {
		display: flex;
	}
	
    .box {
		flex: 1; /* This makes both boxes take equal width */
		margin-right: 10px; /* Optional margin between the boxes */
  	}


  </style>
