//! 图片来源的轻量校验与 OpenAI data URL 投影；不执行 I/O 或解码图片像素。

use base64::Engine;

use super::model_error;
use crate::{AgentError, ContentPart, ImageSource};

pub(crate) fn has_images(parts: &[ContentPart]) -> bool {
    parts
        .iter()
        .any(|part| matches!(part, ContentPart::Image { .. }))
}

pub(crate) fn validate_image(source: &ImageSource) -> Result<(), AgentError> {
    match source {
        ImageSource::Url { url } => {
            let parsed = reqwest::Url::parse(url).map_err(|_| model_error("image_invalid_url"))?;
            if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
                return Err(model_error("image_url_requires_http_or_https"));
            }
        }
        ImageSource::Base64 { media_type, data } => {
            if !matches!(
                media_type.as_str(),
                "image/jpeg" | "image/png" | "image/gif" | "image/webp"
            ) {
                return Err(model_error("image_unsupported_media_type"));
            }
            if data.is_empty()
                || base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .is_err()
            {
                return Err(model_error("image_invalid_base64"));
            }
        }
    }
    Ok(())
}

#[cfg(feature = "openai")]
pub(crate) fn image_url(source: &ImageSource) -> Result<String, AgentError> {
    validate_image(source)?;
    Ok(match source {
        ImageSource::Url { url } => url.clone(),
        ImageSource::Base64 { media_type, data } => format!("data:{media_type};base64,{data}"),
    })
}
