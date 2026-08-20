from dataclasses import dataclass


@dataclass(frozen=True)
class RequestContext:
    user_id: str
    tenant_id: str
    roles: set[str]
    department_id: str | None = None

    def can_access_department(self, department_id: str | None) -> bool:
        if department_id is None:
            return True
        if "admin" in self.roles or "teacher" in self.roles:
            return True
        return self.department_id == department_id


def build_row_level_filter(context: RequestContext) -> dict:
    visibility = ["public"]
    if "student" in context.roles:
        visibility.append("student")
    if "teacher" in context.roles:
        visibility.append("teacher")
    if "admin" in context.roles:
        visibility.append("internal")

    return {
        "tenant_id": context.tenant_id,
        "visibility": visibility,
        "department_id": context.department_id,
    }

