import {LLM_BASE_URL, LLM_API_KEY, LLM_NAME} from '$env/static/private';
import OpenAI from 'openai';

export const  prompt = "You are designed to be helpful while providing only factual information. If you are uncertain, state it and explain why. Give an answer based on information in the following paragraphs.";

export const exampleQuestions = [
	"How to improve data accuracy?",
	"Ensuring compliance with industry standards?",
	"Optimizing research process for efficiency?"
];

export const exampleWarnings = [ "Warning 1", 
					"Warning 2",
					"Warning 3"
				 ];

export const openAI = new OpenAI({
	baseURL: LLM_BASE_URL + "/v1",
	apiKey: LLM_API_KEY ?? ''
});


