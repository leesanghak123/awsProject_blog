<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>

<%@ include file="../layout/header.jsp"%>

<div class="container">

	<form>
		<div class="form-group">
			<input type="text" class="form-control" placeholder="Enter title" id="title" required>
		</div>

		<div class="form-group">
			<textarea class="form-control summernote" rows="5" id="content" required></textarea>
		</div>
	</form>
	<div class="text-right">
		<button id="btn-save" class="btn btn-dark">저장</button>
	</div>
</div>

<script>
	$('.summernote').summernote({
		tabsize : 2,
		height : 300
	});
</script>
<script src="/js/board.js"></script>

<%@ include file="../layout/footer.jsp"%>