fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        embed_resource::compile_for_tests("app.manifest", embed_resource::NONE)
            .manifest_required()
            .expect("failed to embed common-controls manifest into tests");
    }
}
