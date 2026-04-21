use wasm_bindgen::{JsCast, JsValue as WbJsValue};

use leptos::prelude::*;

/// API Documentation page using Redoc (OpenAPI 3.1.0 viewer).
#[component]
pub fn ApiDocs() -> impl IntoView {
    let loading = RwSignal::new(true);
    let error = RwSignal::new(Option::<String>::None);

    // Initialize Redoc on mount by creating a <redoc> element with spec URL.
    Effect::new(move |_| {
        wasm_bindgen_futures::spawn_local(async move {
            if let Some(window) = web_sys::window() {
                // Get the Redoc constructor from window using raw wasm_bindgen.
                let redoc_val = js_sys::Reflect::get(
                    &wasm_bindgen::JsValue::from(&window),
                    &WbJsValue::from_str("Redoc"),
                )
                .ok();
                let redoc = match redoc_val {
                    Some(r) if !r.is_undefined() => r,
                    _ => {
                        error.set(Some("Failed to load Redoc library".to_string()));
                        loading.set(false);
                        return;
                    }
                };

                // Create a <div> container for Redoc.
                let container = web_sys::window()
                    .and_then(|w| w.document())
                    .and_then(|doc| doc.create_element("div").ok())
                    .expect("failed to create div");
                container.set_class_name("api-docs-container");

                // Get the document body and append our container.
                if let Some(body) = window.document().and_then(|d| d.body()) {
                    let _ = body.append_child(&container);
                }

                // Configure Redoc: use spec URL + embedded options for clean display.
                let config = js_sys::Object::new();
                js_sys::Reflect::set(
                    &config,
                    &WbJsValue::from_str("specUrl"),
                    &WbJsValue::from_str("/koji/v1/docs"),
                )
                .unwrap();
                js_sys::Reflect::set(
                    &config,
                    &WbJsValue::from_str("hideHostname"),
                    &WbJsValue::from_bool(true),
                )
                .unwrap();
                js_sys::Reflect::set(
                    &config,
                    &WbJsValue::from_str("disableSearch"),
                    &WbJsValue::from_bool(true),
                )
                .unwrap();
                js_sys::Reflect::set(
                    &config,
                    &WbJsValue::from_str("onlyRequiredInSamples"),
                    &WbJsValue::from_bool(false),
                )
                .unwrap();
                js_sys::Reflect::set(
                    &config,
                    &WbJsValue::from_str("pathInMiddlePanel"),
                    &WbJsValue::from_bool(true),
                )
                .unwrap();
                js_sys::Reflect::set(
                    &config,
                    &WbJsValue::from_str("hideDownloadButton"),
                    &WbJsValue::from_bool(true),
                )
                .unwrap();

                // Call Redoc.init(containerElement, config).
                let init_fn_val = js_sys::Reflect::get(&redoc, &WbJsValue::from_str("init")).ok();
                if let Some(init_fn) =
                    init_fn_val.and_then(|v| v.dyn_into::<js_sys::Function>().ok())
                {
                    if let Err(e) = init_fn.call2(&redoc, &container, &config) {
                        error.set(Some(format!("Redoc init failed: {e:?}")));
                    }
                } else {
                    error.set(Some("Redoc.init not found".to_string()));
                }
            } else {
                error.set(Some("No window available".to_string()));
            }
            loading.set(false);
        });
    });

    view! {
        <div class="page api-docs-page">
            <h1 class="page__title">"API Documentation"</h1>
            <p class="api-docs-subtitle">
                "Interactive reference for the Koji Web API (OpenAPI 3.1.0). "
            </p>

            {move || loading.get().then(|| view! {
                <div class="api-docs-loading">
                    <div class="spinner" />
                    "Loading API documentation..."
                </div>
            })}

            {move || error.get().map(|e| view! {
                <div class="error-banner">{e}</div>
            })}
        </div>
    }
}
