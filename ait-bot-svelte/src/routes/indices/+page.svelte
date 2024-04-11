<script>
	import { json } from '@sveltejs/kit';
	import { onMount } from 'svelte';

	/** @type {import('./$types').PageData} */
	let background = "";
	let query = '';
	let OPENAIMODEL = "mixtral";
	let ENDPOINT = "http://10.103.251.104:8040/v1";
	let MARQO_ENDPOINT = "http://10.103.251.100:8882";
	let APIKEY = "N/A";
	let INDEX = "animal-facts"

// To store the list of objects
/**
 * @type {any[]}
 */
let objects = []; 

  // Function to list objects in a bucket
  async function listObjects() {
    try {
        const response = await fetch("api/store", {
		  method: 'POST',
		  headers: {
			'Content-Type': 'application/json',
			// 'Authorization': "Bearer N/A"
		  }
		});

		let ans = await response.json();
		console.log(ans);
		objects = ans.map(obj => obj.name);
	}
    catch (error){
        console.log(error)
    }
}

let contents = [];
const showBucketContents = async (bucketName) => {
	try {
		const response = await fetch("api/store/contents", {
		  method: 'POST',
		  headers: {
			'Content-Type': 'application/json',
			// 'Authorization': "Bearer N/A"
		  },
		  body: JSON.stringify(bucketName)
		});
		console.log(response);
		const stream = await response;
		console.log(stream);

		// Minio List Objects returns a stream containing all objects in a bucket
		stream.on("data", obj => {
			contents.push(obj);
		})

		stream.on('error', err => {
            console.log(err);
        });

        stream.on('end', () => {
            console.log("Finished reading stream. ");
        });
}
	catch (error) {
		console.log(error);
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
			<ul>
			{#each objects as object}
				<li on:click={() => showBucketContents(object)}>{object}</li>
			{/each}
			</ul>
		</div>

		<div class="box">
			<ul>
			{#each contents as content}
				<li> Here {content} </li>
			{/each}
			</ul>
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
