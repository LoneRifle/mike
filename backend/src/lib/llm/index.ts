import { ProviderV3 } from "@ai-sdk/provider";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { AsyncIterableStream, generateText, jsonSchema, ModelMessage, streamText, TextPart, TextStreamPart, ToolCallPart, ToolResultPart, ToolSet } from "ai";

import { streamClaude, completeClaudeText } from "./claude";
import { streamOpenAI, completeOpenAIText } from "./openai";
import { providerForModel } from "./models";
import type { NormalizedToolCall, NormalizedToolResult, Provider, StreamChatParams, StreamChatResult, UserApiKeys } from "./types";

export * from "./types";
export * from "./models";

type ModelProviderConfig = {
    modelProviderFactory: ({ apiKey }: { apiKey: string }) => ProviderV3;
    defaultApiKey: string;
    keyEnvVar: string;
}

const MODEL_PROVIDER_CONFIGS: Record<Provider, ModelProviderConfig> = {
    gemini: {
        modelProviderFactory: createGoogleGenerativeAI,
        defaultApiKey: process.env.GEMINI_API_KEY?.trim() ?? "",
        keyEnvVar: "GEMINI_API_KEY",
    },
    claude: {
        modelProviderFactory: () => { throw new Error("Unsupported"); },
        defaultApiKey: process.env.ANTHROPIC_API_KEY?.trim() ?? "",
        keyEnvVar: "ANTHROPIC_API_KEY",
    },
    openai: {
        modelProviderFactory: () => { throw new Error("Unsupported"); },
        defaultApiKey: process.env.OPENAI_API_KEY?.trim() ?? "",
        keyEnvVar: "OPENAI_API_KEY",
    },
}

const THINKING_CONFIG = {
    google: {
        // When thinking enabled, ask Gemini to surface thought summaries.
        thinkingConfig:  { includeThoughts: true }
    },
};

const NON_THINKING_CONFIG = {
    google: {
        // When thinking disabled, explicitly zero the thinking budget so the
        // model skips thinking entirely (saves tokens and latency
        // for bulk extraction jobs).
        thinkingConfig: { thinkingBudget: 0 },
    },
};

export async function streamChatWithTools(
    params: StreamChatParams,
): Promise<StreamChatResult> {
    const provider = providerForModel(params.model);
    if (provider === "claude") return streamClaude(params);
    if (provider === "openai") return streamOpenAI(params);

    // Make a copy of input messages so that we inject responses later
    const messages = [...params.messages] as ModelMessage[];
    const model = resolveModel(params, provider);
    const result = streamText({
        model,
        messages,
        system: params.systemPrompt,
        tools: (params.tools ?? []).reduce((toolSet, tool) => {
            toolSet[tool.function.name] = {
                description: tool.function.description,
                inputSchema: jsonSchema(tool.function.parameters),
            };
            return toolSet;   
        }, {} as ToolSet),
        providerOptions: params.enableThinking ? THINKING_CONFIG : NON_THINKING_CONFIG,
    });

    const { assistantParts, toolCalls, fullText } = await handleStreamEvents(params, result.fullStream);

    messages.push({ role: "assistant", content: assistantParts });
    if (toolCalls.length && params.runTools) {
        const toolResults = await params.runTools(toolCalls);
        const toToolResult = (r: NormalizedToolResult): ToolResultPart => {
            const toolName = toolCalls.find((c) => c.id === r.tool_use_id)?.name ?? "tool";
            return {
                type: "tool-result",
                toolName,
                toolCallId: r.tool_use_id,
                output: {
                    type: "text",
                    value: r.content,
                },
            };
        }
        messages.push({ role: "tool", content: toolResults.map(toToolResult) });
    }
    return { fullText };
}

async function handleStreamEvents(params: StreamChatParams, fullStream: AsyncIterableStream<TextStreamPart<ToolSet>>) {
    let fullText = "";
    const toolCalls = [] as NormalizedToolCall[];
    const assistantParts = [] as (ToolCallPart | TextPart)[];
    const { callbacks } = params;
    for await (const event of fullStream) {
        switch (event.type) {
            case "reasoning-delta":
                callbacks?.onReasoningDelta?.(event.text);
                break;
            case "reasoning-end":
                callbacks?.onReasoningBlockEnd?.();
                break;
            case "text-delta":
                fullText += event.text;
                assistantParts.push({ type: "text", text: event.text });
                callbacks?.onContentDelta?.(event.text);
                break;
            case "tool-call":
                assistantParts.push(event);
                const toolCall = {
                    id: event.toolCallId,
                    name: event.toolName,
                    input: event.input,
                };
                toolCalls.push(toolCall);
                callbacks?.onToolCallStart?.(toolCall);
        };
    }
    return { assistantParts, toolCalls, fullText };
}

export async function completeText(params: {
    model: string;
    systemPrompt?: string;
    user: string;
    maxTokens?: number;
    apiKeys?: UserApiKeys;
}): Promise<string> {
    const provider = providerForModel(params.model);
    if (provider === "claude") return completeClaudeText(params);
    if (provider === "openai") return completeOpenAIText(params);

    const model = resolveModel(params, provider);
    const result = await generateText({
        model,
        system: params.systemPrompt,
        messages: [{ role: "user", content: params.user }],
    });
    return result.text;
}

function resolveModel(params: { model: string; apiKeys?: UserApiKeys; }, provider: Provider) {
    const { defaultApiKey, modelProviderFactory, keyEnvVar } = MODEL_PROVIDER_CONFIGS[provider];
    const apiKey = (params.apiKeys ?? {})[provider] ?? defaultApiKey;
    if (!apiKey) {
        throw new Error(
            `API key for ${provider} is not configured. Set ${keyEnvVar} or add a user key.`
        );
    }
    const modelProvider = modelProviderFactory({ apiKey });

    return modelProvider.languageModel(params.model);
}

