<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>

<%@ include file="../layout/header.jsp"%>

<div class="container">
	<div>
		<input type="hidden" id="id" name="id" value="${gpt.id}"> <strong>출발지: <span>${gpt.start}</span></strong><br> <strong>목적지: <span>${gpt.end}</span></strong><br> <strong>여행
			기간: <span>${gpt.days}일</span>
		</strong>
	</div>
	<hr />
	<div>
		<textarea class="form-control summernote" rows="5" id="content">${gpt.result}</textarea>
	</div>
	<div class="text-right mt-3">
		<button id="btn-plan-update" class="btn btn-dark">글 수정 완료</button>
	</div>
</div>

<script>
	$('.summernote').summernote({
		tabsize : 2,
		height : 300
	});
</script>
<script src="/js/user.js"></script>
<%@ include file="../layout/footer.jsp"%>