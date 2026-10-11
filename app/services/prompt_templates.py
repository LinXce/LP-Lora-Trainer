"""User-owned prompt text. Tasks use snapshots, never live template references."""
import uuid
from adapters.sampling import prompt_error
from app.services.common import ServiceError, now


class PromptTemplateService:
    def __init__(self, store):
        self.store = store

    def list(self):
        return sorted(self.store.list("prompt_template"), key=lambda item: item["updated_at"], reverse=True)

    def get(self, key):
        item = self.store.get("prompt_template", key)
        if item is None:
            raise ServiceError("提示词模板不存在", 404)
        return item

    def save(self, name, prompt, key=None):
        name, prompt = name.strip(), prompt.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not name or len(name) > 200:
            raise ServiceError("模板名称不能为空且不能超过 200 字符", 422)
        if not prompt:
            raise ServiceError("提示词内容不能为空", 422)
        error = prompt_error(prompt)
        if error:
            raise ServiceError(error, 422)
        previous = self.get(key) if key is not None else None
        key = key or uuid.uuid4().hex
        stamp = now()
        return self.store.put("prompt_template", key, dict(template_id=key, name=name, prompt=prompt,
            created_at=previous["created_at"] if previous else stamp, updated_at=stamp))

    def delete(self, key):
        self.get(key)
        self.store.delete("prompt_template", key)
