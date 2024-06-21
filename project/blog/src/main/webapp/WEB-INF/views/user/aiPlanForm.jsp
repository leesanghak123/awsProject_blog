<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>

<%@ include file="../layout/header.jsp"%>

<div class="container">
	<h1>AI 여행계획</h1>
	<form>
		<input type="hidden" id="id" value="${principal.user.id}" />
		<div class="input-group mb-3 input-group-sm">
			<div class="input-group-prepend">
				<span class="input-group-text">목적지</span>
			</div>
			<input type="text" id="stage_add" placeholder="목적지 장소를 입력해주세요" class="form-control">
		</div>
		<div class="text-right">
			<button type="button" id="btn-write" class="btn btn-dark">작성</button>
			<button type="button" id="btn-plan-next" class="btn btn-dark" disabled>다른 계획</button>
		</div>
		<hr>
		<div class="form-group">
			<label for="result">AI 여행계획 결과:</label>
			<textarea class="form-control" id="result" rows="10"></textarea>
		</div>
		<div class="text-right">
			<button type="button" id="btn-plan-save" class="btn btn-dark">저장</button>
		</div>
	</form>
</div>


<script src="/js/user.js"></script>
<%@ include file="../layout/footer.jsp"%>