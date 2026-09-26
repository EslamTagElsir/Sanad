// Sanad on Cloudflare Containers.
//
// The Worker forwards every request to one container running the existing
// Dockerfile (FastAPI + embedding model). The container has no D1 credentials:
// it calls http://d1.sanad/query, and the outbound handler below runs those
// statements on D1 through the binding. Only follow-up tickets live in D1
// (see stage4_production/ticket_store.py).
import { Container, getContainer } from "@cloudflare/containers";
import { env } from "cloudflare:workers";

export class SanadContainer extends Container {
	defaultPort = 7860;
	// New (non follow-up) tickets live on the container's disk and are lost when
	// it sleeps — an accepted trade-off. Follow-ups are safe in D1.
	sleepAfter = "30m";
	envVars = {
		OPENROUTER_API_KEY: env.OPENROUTER_API_KEY ?? "",
		OPENROUTER_MODELS: env.OPENROUTER_MODELS ?? "",
		SANAD_JWT_SECRET: env.SANAD_JWT_SECRET ?? "",
		SANAD_USERS_JSON: env.SANAD_USERS_JSON ?? "",
		SENDGRID_API_KEY: env.SENDGRID_API_KEY ?? "",
		SENDGRID_FROM_EMAIL: env.SENDGRID_FROM_EMAIL ?? "",
		SANAD_PUBLIC_URL: env.SANAD_PUBLIC_URL ?? "",
		SANAD_D1_URL: "http://d1.sanad/query",
	};
}

// Body: {"statements": [{"sql": "...", "params": [...]}, ...]}
// Runs them as one D1 batch (a single transaction) and returns, per statement,
// {"rows": [...], "changes": n} — the shape ticket_store._D1Store expects.
SanadContainer.outboundByHost = {
	"d1.sanad": async (request, env) => {
		if (request.method !== "POST" || new URL(request.url).pathname !== "/query") {
			return new Response("Not found", { status: 404 });
		}
		try {
			const { statements } = await request.json();
			const results = await env.DB.batch(
				statements.map(({ sql, params }) => env.DB.prepare(sql).bind(...(params ?? []))),
			);
			return Response.json(results.map((r) => ({ rows: r.results ?? [], changes: r.meta?.changes ?? 0 })));
		} catch (err) {
			return new Response(`D1 error: ${err}`, { status: 500 });
		}
	},
};

export default {
	async fetch(request, env) {
		// One instance: the in-memory caches and login lockout assume a single process.
		return getContainer(env.SANAD, "main").fetch(request);
	},
};
