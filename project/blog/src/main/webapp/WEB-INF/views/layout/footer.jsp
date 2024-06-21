<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<br />
<div id="foot" class="jumbotron text-center" style="margin-bottom: 0">
	<p>@Create by Sang</p>
	<p>📞010-0000-0000</p>
	<p>🏴부산광역시 남구 XX동</p>
</div>

<!-- footer 아래 고정 -->
<style>
html, body {
    height: 100%;
    margin: 0;
    display: flex;
    flex-direction: column;
}

.container {
    flex-grow: 1;
}

#foot {
            margin-top: auto; /* Flexbox를 사용하여 footer를 화면 하단으로 이동 */
            padding: 2vh; /* 패딩을 줄여 크기를 줄임 */
            font-size: 0.8em;   /* 글자 크기를 줄임 */
        }
</style>

</body>
</html>

