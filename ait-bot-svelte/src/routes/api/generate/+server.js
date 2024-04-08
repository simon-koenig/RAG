import {json} from '@sveltejs/kit'; 

// POST Route api for marqo index search

/** @type {import('./$types').RequestHandler} */
export async function POST({request}) {
	let ENDPOINT = "http://10.103.251.104:8040/v1";
  const llmData = await request.json();

  console.log("Hi")
  console.log(ENDPOINT);
  console.log(JSON.stringify(llmData))

  try {
    const response = await fetch(ENDPOINT + '/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(llmData)
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();
    return json(data);
  } catch (error) {
    console.error('Error fetching data:', error);
    return json(error);
  }
}
