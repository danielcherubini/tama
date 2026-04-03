use leptos::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PullResponse {
    job_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PullStatus {
    status: String,
    file_name: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ValidationError {
    error: String,
    available_quants: Option<Vec<String>>,
}

#[component]
pub fn Pull() -> impl IntoView {
    let repo_id = RwSignal::new(String::new());
    let quant = RwSignal::new(String::new());
    let job_id = RwSignal::new(Option::<String>::None);
    let poll_trigger = RwSignal::new(0u32);
    let job_status = RwSignal::new(Option::<PullStatus>::None);
    let error_msg = RwSignal::new(Option::<String>::None);
    let available_quants = RwSignal::new(Option::<Vec<String>>::None);

    // Poll job status when we have a job_id
    let poll_resource = LocalResource::new(move || async move {
        let _ = poll_trigger.get(); // track trigger
        let id = job_id.get()?;
        let resp = gloo_net::http::Request::get(&format!("/kronk/v1/pulls/{}", id))
            .send()
            .await
            .ok()?;
        resp.json::<PullStatus>().await.ok()
    });

    // Effect to update job_status from poll resource and schedule next poll
    Effect::new(move |_| {
        if let Some(guard) = poll_resource.get() {
            if let Some(status) = guard.take() {
                let done = status.status == "done" || status.status == "error";
                job_status.set(Some(status));
                if !done {
                    let trigger = poll_trigger;
                    wasm_bindgen_futures::spawn_local(async move {
                        gloo_timers::future::TimeoutFuture::new(1_000).await;
                        trigger.update(|n| *n += 1);
                    });
                }
            }
        }
    });

    let submit: Action<(), (), LocalStorage> = Action::new_unsync(move |_: &()| {
        let repo = repo_id.get();
        let q = quant.get();
        async move {
            error_msg.set(None);
            available_quants.set(None);
            let body = serde_json::json!({ "repo_id": repo, "quant": q });
            let send_result = gloo_net::http::Request::post("/kronk/v1/pulls")
                .json(&body)
                .map(|r| r.send());
            match send_result {
                Ok(fut) => match fut.await {
                    Ok(resp) => {
                        let status = resp.status();
                        if status == 200 || status == 201 || status == 202 {
                            if let Ok(pr) = resp.json::<PullResponse>().await {
                                job_id.set(Some(pr.job_id));
                                poll_trigger.update(|n| *n += 1);
                            }
                        } else if status == 422 {
                            if let Ok(ve) = resp.json::<ValidationError>().await {
                                error_msg.set(Some(ve.error));
                                available_quants.set(ve.available_quants);
                            }
                        } else {
                            error_msg.set(Some(format!("Request failed with status {}", status)));
                        }
                    }
                    Err(e) => {
                        error_msg.set(Some(format!("Request error: {}", e)));
                    }
                },
                Err(e) => {
                    error_msg.set(Some(format!("Failed to build request: {}", e)));
                }
            }
        }
    });

    view! {
        <h1>"Pull Model"</h1>
        <div>
            <label>"Repo ID: "</label>
            <input
                type="text"
                prop:value=move || repo_id.get()
                on:input=move |e| repo_id.set(event_target_value(&e))
                placeholder="e.g. meta-llama/Llama-3.2-1B"
            />
        </div>
        <div>
            <label>"Quant: "</label>
            <input
                type="text"
                prop:value=move || quant.get()
                on:input=move |e| quant.set(event_target_value(&e))
                placeholder="e.g. Q4_K_M"
            />
        </div>
        <button on:click=move |_| { submit.dispatch(()); }>"Pull"</button>

        {move || error_msg.get().map(|e| view! {
            <p style="color: red">"Error: " {e}</p>
        })}

        {move || available_quants.get().map(|quants| view! {
            <div>
                <p>"Available quants:"</p>
                <ul>
                    {quants.into_iter().map(|q| view! { <li>{q}</li> }).collect::<Vec<_>>()}
                </ul>
            </div>
        })}

        {move || job_status.get().map(|s| view! {
            <div>
                <p>"Status: " {s.status}</p>
                {s.file_name.map(|f| view! { <p>"File: " {f}</p> })}
                {s.error.map(|e| view! { <p style="color: red">"Error: " {e}</p> })}
            </div>
        })}
    }
}
