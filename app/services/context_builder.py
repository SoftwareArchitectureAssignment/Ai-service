import logging
from app.interfaces.context_builder import IContextBuilder

logger = logging.getLogger(__name__)


class ContextBuilder(IContextBuilder):
    
    def build_rag_context(self, docs: list) -> str:
        if not docs:
            return ""
        
        return "\n\n".join([doc.page_content for doc in docs])
    
    def build_context_with_metadata(self, docs: list) -> str:
        if not docs:
            return ""
        
        context_parts = []
        for doc in docs:
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            course_uid = metadata.get('course_uid', 'unknown')
            context_parts.append(f"{doc.page_content}\nCourse UID: {course_uid}")
        
        return "\n\n".join(context_parts)
