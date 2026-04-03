use leptos::prelude::*;
use leptos_router::components::A;

#[component]
pub fn Nav() -> impl IntoView {
    view! {
        <nav>
            <A href="/">"Dashboard"</A>
            " | "
            <A href="/models">"Models"</A>
            " | "
            <A href="/pull">"Pull Model"</A>
        </nav>
    }
}
