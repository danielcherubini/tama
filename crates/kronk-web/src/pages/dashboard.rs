use leptos::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SystemHealth {
    status: String,
    service: String,
    models_loaded: u32,
    vram: Option<VramInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VramInfo {
    used_mib: i64,
    total_mib: i64,
}

#[component]
pub fn Dashboard() -> impl IntoView {
    let health = LocalResource::new(|| async move {
        let resp = gloo_net::http::Request::get("/kronk/v1/system/health")
            .send()
            .await
            .ok()?;
        resp.json::<SystemHealth>().await.ok()
    });

    let restart: Action<(), (), LocalStorage> = Action::new_unsync(|_: &()| async move {
        let _ = gloo_net::http::Request::post("/kronk/v1/system/restart")
            .send()
            .await;
    });

    view! {
        <h1>"Dashboard"</h1>
        <Suspense fallback=|| view! { <p>"Loading..."</p> }>
            {move || {
                health.get().map(|h| {
                    // Deref the SendWrapper to get the inner Option<SystemHealth>
                    let h = h.take();
                    match h {
                        Some(h) => view! {
                            <p>"Status: " {h.status}</p>
                            <p>"Models loaded: " {h.models_loaded}</p>
                            {h.vram.map(|v| view! {
                                <p>"VRAM: " {v.used_mib} " / " {v.total_mib} " MiB"</p>
                            })}
                        }.into_any(),
                        None => view! { <p>"Failed to load health data"</p> }.into_any(),
                    }
                })
            }}
        </Suspense>
        <button on:click=move |_| { restart.dispatch(()); }>"Restart Kronk"</button>
    }
}
