<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>

<%@ include file="../layout/header.jsp"%>

<div class="container">
	
	<div class="text-right">
		<button class="btn btn-dark" onclick="history.back()">돌아가기</button>
			<a href="/user/planUpdate/${gpt.id}" class="btn btn-warning">수정</a>
			<button id="btn-plan-delete" class="btn btn-warning">삭제</button>
	</div>
	
	<div>
		<input type="hidden" id="id" value="${gpt.id}">
		목적지: <span>${gpt.stage}</span><br>
	</div>
	<br />
	<hr />
	<div>
		<div>${gpt.result}</div>
	</div>
	<hr />
</div>


<script src="/js/user.js"></script>
<%@ include file="../layout/footer.jsp"%>