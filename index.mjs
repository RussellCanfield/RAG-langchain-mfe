import "@tensorflow/tfjs-node";
import { YoutubeLoader } from "langchain/document_loaders/web/youtube";
import { PlaywrightWebBaseLoader } from "langchain/document_loaders/web/playwright";
import { OpenAI } from "langchain/llms/openai";
import { loadQAMapReduceChain } from "langchain/chains";

const youtubeVideoSources = [
	"https://youtu.be/bXfK873ASzY",
	"https://www.youtube.com/watch?v=w58aZjACETQ&ab_channel=JackHerrington",
	"https://www.youtube.com/watch?v=D3XYAx30CNc&ab_channel=JackHerrington",
	"https://www.youtube.com/watch?v=x22F4hSdZJM&ab_channel=JackHerrington",
	"https://www.youtube.com/watch?v=2eGXIbc6lZA&ab_channel=ZackJackson",
	"https://www.youtube.com/watch?v=AU7dKWNfWiA&ab_channel=JackHerrington",
	"https://www.youtube.com/watch?v=-ei6RqZilYI&ab_channel=Pusher",
];

const youtubeLoaders = youtubeVideoSources.map((videoURL) =>
	YoutubeLoader.createFromUrl(videoURL, {
		language: "en",
		addVideoInfo: true,
	}).load()
);

const webLoader = new PlaywrightWebBaseLoader(
	"https://medium.com/swlh/webpack-5-module-federation-a-game-changer-to-javascript-architecture-bcdd30e02669"
).load();

const loaders = await Promise.all([...youtubeLoaders, webLoader]);

const docs = loaders[0];

const model = new OpenAI({
	modelName: "gpt-4",
	temperature: 0,
	maxConcurrency: 10,
});

const chain = loadQAMapReduceChain(model);

const res = await chain.call({
	input_documents: docs,
	question: "What is Module Federation and how can I use it?",
});

console.log({ res });
